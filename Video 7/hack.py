n = input()
l = len(n)

for i in range(0,l,2):
    if int(n[i])%2==0:
        even = True
for j in range(1,l,2):
    if n[j] in "2 3 5 7":
        odd = True
if odd and even:
    print("Good")
