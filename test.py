import csv
f = open('titles.csv', 'r', encoding='utf-8')
reader = csv.reader(f)
header = next(reader)
runtimeidx = header.index('runtime')
maxruntime = 0
for row in reader:
    runtime = float(row[runtimeidx])
    if runtime > maxruntime:
        maxruntime = runtime
print("Maximum runtime: ",maxruntime," minutes")
f.close()
