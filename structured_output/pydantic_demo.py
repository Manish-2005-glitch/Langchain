from pydantic import BaseModel, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'Manish'
    age : Optional[int] = None
    cgpa : float = Field(gt = 0, lt = 10, default = 5.0, description = "A decimal value representing the cgpa of the student")
    
new_student = {'age' : 22}

student = Student(**new_student)
print(student)