
class A:
    def __init__(self):
        self.b = 1

    def do(self):
        print('Class attribute "b" is', self.b)

a = A()
a.do()