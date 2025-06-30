#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_processor import (
    StudentPerformanceDataProcessor,
    load_and_prepare_data,
    compute_denormalized_metrics
)

class StudentSequenceDataset(Dataset):
    def __init__(self, student_sequences: List[Dict], max_courses_per_semester: int = 15, 
                 max_semesters: int = 10):
        self.sequences = student_sequences
        self.max_courses = max_courses_per_semester
        self.max_semesters = max_semesters
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        course_features = torch.zeros(self.max_semesters, self.max_courses, 7)
        semester_features = torch.zeros(self.max_semesters, 11)
        course_masks = torch.zeros(self.max_semesters, self.max_courses)
        semester_masks = torch.zeros(self.max_semesters)
        
        for sem_idx, sem_data in enumerate(seq['semesters'][:self.max_semesters]):
            semester_features[sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
            semester_masks[sem_idx] = 1.0
            
            courses = sem_data['courses'][:self.max_courses]
            for course_idx, course in enumerate(courses):
                course_features[sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
                course_masks[sem_idx, course_idx] = 1.0
        
        target_gpa = seq['target_gpa']
        target_cpa = seq['target_cpa']
        
        return {
            'course_features': course_features,
            'semester_features': semester_features, 
            'course_masks': course_masks,
            'semester_masks': semester_masks,
            'target_gpa': torch.tensor(target_gpa, dtype=torch.float32),
            'target_cpa': torch.tensor(target_cpa, dtype=torch.float32)
        }

class CourseEncoder(nn.Module):
    def __init__(self, course_feature_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.course_embedding = nn.Sequential(
            nn.Linear(course_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, course_features, course_masks):
        batch_size, max_sems, max_courses, feat_dim = course_features.shape
        
        course_features_flat = course_features.view(-1, feat_dim)
        course_embeddings = self.course_embedding(course_features_flat)
        course_embeddings = course_embeddings.view(batch_size, max_sems, max_courses, -1)
        
        masked_embeddings = course_embeddings * course_masks.unsqueeze(-1)
        semester_course_sum = masked_embeddings.sum(dim=2)
        
        course_counts = course_masks.sum(dim=2, keepdim=True)
        semester_representations = semester_course_sum / (course_counts + 1e-8)
        
        return semester_representations

class SemesterTransformerEncoder(nn.Module):
    def __init__(self, semester_dim: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(50, semester_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=semester_dim,
            nhead=num_heads,
            dim_feedforward=semester_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, semester_representations, semester_masks):
        seq_len = semester_representations.size(1)
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        
        enhanced_representations = semester_representations + pos_enc
        
        attention_mask = ~semester_masks.bool()
        
        transformed = self.transformer(enhanced_representations, src_key_padding_mask=attention_mask)
        
        return transformed

class StudentPerformancePredictor(nn.Module):
    def __init__(self, course_feature_dim: int = 7, semester_feature_dim: int = 11, 
                 hidden_dim: int = 64, lstm_hidden: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.course_encoder = CourseEncoder(course_feature_dim, hidden_dim)
        
        self.semester_feature_proj = nn.Linear(semester_feature_dim, hidden_dim)
        
        self.semester_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_encoder = SemesterTransformerEncoder(hidden_dim, num_heads, 2)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, course_features, semester_features, course_masks, semester_masks):
        course_representations = self.course_encoder(course_features, course_masks)
        
        semester_proj = self.semester_feature_proj(semester_features)
        
        fused_representations = self.semester_fusion(
            torch.cat([course_representations, semester_proj], dim=-1)
        )
        
        transformer_output = self.transformer_encoder(fused_representations, semester_masks)
        
        lstm_output, _ = self.lstm(transformer_output)
        
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output,
                                       key_padding_mask=~semester_masks.bool())
        
        valid_lengths = semester_masks.sum(dim=1).long()
        batch_indices = torch.arange(attn_output.size(0))
        last_valid_outputs = attn_output[batch_indices, valid_lengths - 1]
        
        predictions = self.predictor(last_valid_outputs)
        
        return predictions

class ModelTrainer:
    def __init__(self, model: nn.Module, processor: StudentPerformanceDataProcessor = None, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.processor = processor
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, min_lr=1e-6)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            course_features = batch['course_features'].to(self.device)
            semester_features = batch['semester_features'].to(self.device)
            course_masks = batch['course_masks'].to(self.device)
            semester_masks = batch['semester_masks'].to(self.device)
            target_gpa = batch['target_gpa'].to(self.device)
            target_cpa = batch['target_cpa'].to(self.device)
            
            targets = torch.stack([target_gpa, target_cpa], dim=1)
            
            self.optimizer.zero_grad()
            predictions = self.model(course_features, semester_features, course_masks, semester_masks)
            
            if torch.isnan(predictions).any():
                print("Warning: NaN detected in predictions")
                continue
                
            loss = self.criterion(predictions, targets)
            
            if torch.isnan(loss):
                print("Warning: NaN detected in loss")
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                course_features = batch['course_features'].to(self.device)
                semester_features = batch['semester_features'].to(self.device)
                course_masks = batch['course_masks'].to(self.device)
                semester_masks = batch['semester_masks'].to(self.device)
                target_gpa = batch['target_gpa'].to(self.device)
                target_cpa = batch['target_cpa'].to(self.device)
                
                targets = torch.stack([target_gpa, target_cpa], dim=1)
                
                predictions = self.model(course_features, semester_features, course_masks, semester_masks)
                
                if torch.isnan(predictions).any():
                    continue
                    
                loss = self.criterion(predictions, targets)
                
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            return {
                'loss': float('inf'),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
            
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        valid_mask = ~np.isnan(all_predictions).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
        valid_predictions = all_predictions[valid_mask]
        valid_targets = all_targets[valid_mask]
        
        if len(valid_predictions) == 0:
            return {
                'loss': total_loss / len(dataloader),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
        
        gpa_mse = mean_squared_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mse = mean_squared_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_mae = mean_absolute_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mae = mean_absolute_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_r2 = r2_score(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_r2 = r2_score(valid_targets[:, 1], valid_predictions[:, 1])
        
        results = {
            'loss': total_loss / len(dataloader),
            'gpa_mse': gpa_mse, 'cpa_mse': cpa_mse,
            'gpa_mae': gpa_mae, 'cpa_mae': cpa_mae,
            'gpa_r2': gpa_r2, 'cpa_r2': cpa_r2
        }
        
        if self.processor and hasattr(self.processor, 'performance_scaler'):
            denormalized_metrics = compute_denormalized_metrics(
                valid_predictions, valid_targets, self.processor
            )
            results.update(denormalized_metrics)
        
        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        train_losses = []
        val_losses = []
        gpa_mse_history = []
        gpa_mae_history = []
        cpa_mse_history = []
        cpa_mae_history = []
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            gpa_mse_history.append(val_metrics['gpa_mse'])
            gpa_mae_history.append(val_metrics['gpa_mae'])
            cpa_mse_history.append(val_metrics['cpa_mse'])
            cpa_mae_history.append(val_metrics['cpa_mae'])
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
            print(f'  GPA - MSE: {val_metrics["gpa_mse"]:.4f}, MAE: {val_metrics["gpa_mae"]:.4f}, R2: {val_metrics["gpa_r2"]:.4f}')
            print(f'  CPA - MSE: {val_metrics["cpa_mse"]:.4f}, MAE: {val_metrics["cpa_mae"]:.4f}, R2: {val_metrics["cpa_r2"]:.4f}')
            
            if 'gpa_mse_denorm' in val_metrics:
                print(f'  [DENORM] GPA - MSE: {val_metrics["gpa_mse_denorm"]:.4f}, MAE: {val_metrics["gpa_mae_denorm"]:.4f}, R2: {val_metrics["gpa_r2_denorm"]:.4f}')
                print(f'  [DENORM] CPA - MSE: {val_metrics["cpa_mse_denorm"]:.4f}, MAE: {val_metrics["cpa_mae_denorm"]:.4f}, R2: {val_metrics["cpa_r2_denorm"]:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'gpa_mse': gpa_mse_history,
            'gpa_mae': gpa_mae_history,
            'cpa_mse': cpa_mse_history,
            'cpa_mae': cpa_mae_history
        }

def plot_training_results(training_history, test_metrics, base_filename='transformer_training'):
    epochs = range(1, len(training_history['train_losses']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, training_history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, training_history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, training_history['gpa_mse'], 'g-', label='GPA MSE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('GPA MSE Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, training_history['cpa_mse'], 'orange', label='CPA MSE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('CPA MSE Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, training_history['gpa_mae'], 'purple', label='GPA MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('GPA MAE Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, training_history['cpa_mae'], 'brown', label='CPA MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('CPA MAE Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    metrics = ['GPA MSE', 'GPA MAE', 'CPA MSE', 'CPA MAE']
    values = [
        test_metrics['gpa_mse'], 
        test_metrics['gpa_mae'], 
        test_metrics['cpa_mse'], 
        test_metrics['cpa_mae']
    ]
    colors = ['red', 'blue', 'orange', 'purple']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Error Value')
    plt.title('Final Test Metrics')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved as {base_filename}.png")

def main():
    parser = argparse.ArgumentParser(description='Transformer Model for Student Performance Prediction')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV file')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_model', type=str, default='best_model.pth', help='Path to save the trained model')
    parser.add_argument('--output_plot', type=str, default='transformer_results.png', help='Path to save the training plot')
    
    args = parser.parse_args()
    
    print("=== TRANSFORMER MODEL FOR STUDENT PERFORMANCE PREDICTION ===")
    print("Architecture: Course Encoder + Transformer + LSTM + Attention")
    print("All metrics computed on original scale (denormalized)")
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA không khả dụng! Sử dụng CPU...")
        device = torch.device('cpu')
    else:
        print(f"\nCUDA khả dụng! GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    
    print(f"\nDevice: {device}")
    
    try:
        sequences, processor = load_and_prepare_data(args.course_csv, args.performance_csv)
        
        train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
        train_sequences, val_sequences = train_test_split(train_sequences, test_size=0.2, random_state=42)
        
        print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
        
        train_dataset = StudentSequenceDataset(train_sequences)
        val_dataset = StudentSequenceDataset(val_sequences)
        test_dataset = StudentSequenceDataset(test_sequences)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                pin_memory=True if device.type == 'cuda' else False)
        
        print("\nInitializing model...")
        model = StudentPerformancePredictor()
        model = model.to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        trainer = ModelTrainer(model, processor, device)
        
        print("\nStarting training...")
        training_history = trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        print("\nEvaluating on test set...")
        trainer.model.load_state_dict(torch.load(args.output_model))
        test_metrics = trainer.evaluate(test_loader)
        
        processor.save_scalers()
        
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"\nGPA Prediction (Normalized):")
        print(f"  - MSE: {test_metrics['gpa_mse']:.4f}")
        print(f"  - MAE: {test_metrics['gpa_mae']:.4f}")
        print(f"  - R²: {test_metrics['gpa_r2']:.4f}")
        print(f"\nCPA Prediction (Normalized):")
        print(f"  - MSE: {test_metrics['cpa_mse']:.4f}")
        print(f"  - MAE: {test_metrics['cpa_mae']:.4f}")
        print(f"  - R²: {test_metrics['cpa_r2']:.4f}")
        
        if 'gpa_mse_denorm' in test_metrics:
            print(f"\n=== RESULTS ON ORIGINAL SCALE (DENORMALIZED) ===")
            print(f"GPA Prediction:")
            print(f"  - MSE: {test_metrics['gpa_mse_denorm']:.4f}")
            print(f"  - MAE: {test_metrics['gpa_mae_denorm']:.4f}")
            print(f"  - R²: {test_metrics['gpa_r2_denorm']:.4f}")
            print(f"\nCPA Prediction:")
            print(f"  - MSE: {test_metrics['cpa_mse_denorm']:.4f}")
            print(f"  - MAE: {test_metrics['cpa_mae_denorm']:.4f}")
            print(f"  - R²: {test_metrics['cpa_r2_denorm']:.4f}")
        
        plot_training_results(training_history, test_metrics, args.output_plot.replace('.png', ''))
        
        print(f"\nFiles saved:")
        print(f"  - Model: '{args.output_model}'")
        print(f"  - Training plot: '{args.output_plot}'")
        print(f"  - Performance scaler: 'feature_scaler.pkl'")
        print(f"  - Course scaler: 'course_scaler.pkl'")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"\nGPU memory cleared")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 