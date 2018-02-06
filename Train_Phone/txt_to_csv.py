import csv
txt_file = r"images/labels.txt"
csv_file = r"images/labels.csv"

in_txt = open(txt_file, "r",encoding="utf8").read()
csvreader=csv.reader(in_txt.splitlines())

row=[]
for r in csvreader:
    row.append(r[0].split(" "))
   
with open(csv_file,'w',newline='') as out:
    mywriter=csv.writer(out)
    mywriter.writerow(['Filename','n_xcord','n_ycord'])
    for rr in row:
        mywriter.writerow(rr)