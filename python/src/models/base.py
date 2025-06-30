from sqlalchemy import Column, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base

class BaseModel(Base):
    __abstract__ = True
    
    createdAt = Column(DateTime(timezone=True), server_default=func.now())
    updatedAt = Column(DateTime(timezone=True), onupdate=func.now())
    isDeleted = Column(Boolean, default=False) 