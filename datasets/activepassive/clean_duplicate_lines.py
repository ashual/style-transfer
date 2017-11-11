lines_seen = set()
outfile = open("clean.txt", "w")
for line in open("input.txt", "r"):
    if line not in lines_seen:
        outfile.write(line)
        lines_seen.add(line)
outfile.close()