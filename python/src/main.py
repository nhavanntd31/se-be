from database import get_db
from services.student_predict_services import StudentPredictService
from services.student_service import StudentService
import sys
import traceback

def main(current_semester_id=None, next_semester_id=None):
    db = None
    try:
        print(f'Starting prediction for current_semester_id: {current_semester_id}, next_semester_id: {next_semester_id}', flush=True)
        db = next(get_db())
        student_predict_service = StudentPredictService(db)
        result = student_predict_service.predict_for_all_students(current_semester_id, next_semester_id)
        print(f'Prediction completed successfully for current_semester_id: {current_semester_id}, next_semester_id: {next_semester_id}', flush=True)
        return result
    except Exception as e:
        error_msg = f'Error in prediction for current_semester_id: {current_semester_id}, next_semester_id: {next_semester_id}: {str(e)}'
        print(error_msg, file=sys.stderr, flush=True)
        print(f'Full traceback: {traceback.format_exc()}', file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if db:
            db.close()

if __name__ == "__main__":
    current_semester_id = sys.argv[1] if len(sys.argv) > 1 else None
    next_semester_id = sys.argv[2] if len(sys.argv) > 2 else None
    main(current_semester_id, next_semester_id)
