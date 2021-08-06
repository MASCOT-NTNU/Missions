class Employee:
	'All the employee'
	empCount = 0

	def __init__(self, name, salary):
		self.name = name
		self.salary = salary
		Employee.empCount += 1

	def displayCount(self):
		print("Total Employee {:02d}".format(Employee.empCount))

	def displayEmployee(self):
		print("Name: ", self.name, ", Salary: ", self.salary)

A = Employee("DJ", 9999999)
A.displayEmployee()
A.displayCount()

B = Employee("YL", 99999999)
B.displayCount()
B.displayEmployee()

class Test:
	def prt(self):
		print(self)
		print(self.__class__)

t = Test()
t.prt()


emp1 = Employee("Zara", 2000)
emp2 = Employee("Manni", 5000)

emp1.age = 7
print(emp1.age)
emp1.age = 8
print(emp1.age)
del emp1.age

print(hasattr(emp1, "age"))
# print(getattr(emp1, "age"))
setattr(emp1, "age", 8)
delattr(emp1, "age")
print("Employee.__doc__: \n", Employee.__doc__)
print("Employee.__name__: \n", Employee.__name__)
print("Employee.__module__: \n", Employee.__module__)
print("Employee.__bases__: \n", Employee.__bases__)
print("Employee.__dict__: \n", Employee.__dict__)

a = 10
print(id(a))






