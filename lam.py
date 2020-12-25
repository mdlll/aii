def my_test(x):
   # 代码3
   print("my_test")
   return x

if __name__ == "__main__":
   f = lambda x : my_test(x) # 代码1
   print(f(2))  # 代码2


'''
f = lambda x:my_test(x)   # 代码1
等价于
def f(x):
	return my_test(x)
'''