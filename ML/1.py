import csv
a=[]
with open("C:\\Users\\PRAJNA\\Documents\\enjoysport.csv",'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)

print("the length of instances: ",len(a))
num_attributes=len(a[0])-1

print("Initial hypothesis: ")
hypothesis=['0']*num_attributes
print(hypothesis)

for i in range(0, len(a)):
    if a[i][num_attributes]=='yes':
        for j in range(0, num_attributes):
            if hypothesis[j]=='0' or hypothesis[j]==a[i][j]:
                hypothesis[j]=a[i][j]
            else:
                hypothesis[j]='?'
    print("the hypothesis for instances {} is\n" .format(i+1),hypothesis)
    
print("maximum hypothesis is: ")
print(hypothesis) 
