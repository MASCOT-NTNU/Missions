import os

os.system("echo | find . | grep -E \"__pycache__\" > .gitignore")
os.system("echo | find . | grep -E \".idea\" >> .gitignore")
os.system("echo | find . | grep -E \".DS_store\" >> .gitignore")
os.system("echo | find . | grep -E \".gitignore\" >> .gitignore")
os.system("echo | find . | grep -E \".git\" >> .gitignore")
os.system("echo | find . | grep -E \"Sigma_sal.txt\" >> .gitignore")
os.system("echo | find . | grep -E \"samples_2020.05.01.nc\" >> .gitignore")
os.system("echo | find . | grep -E \"master.zip\" >> .gitignore")
print("hello world")


