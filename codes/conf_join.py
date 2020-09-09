

def count_conf_join(fPath):
    P = []
    file_list = []
    ids = []
    with open(fPath) as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '')
            items = line.split('\t')
            print(items)
            ids.append(items[0])
            P.append([float(items[1]), float(items[2]), int(items[-1])])
            line = f.readline()
    m = len(P[0])
    n = len(P)
    C = [[0]*(m-1)for i in range(m-1)]
    t = Average(P)
    print(t)
    for i in range(n):
        cnt = 0
        y0 = -1
        p = 0
        for j in range(m-1):
            if P[i][j] >= t[j]:
                cnt+=1
                y0 = j if p <= P[i][j] else y0
                p = max(p, P[i][j])
        y1 = P[i][m-1]
        if y1 == y0: file_list.append(ids[i] + " " + str(P[i][-1]))
        if cnt > 0:
            C[y1][y0] += 1
    print(C)
    with open('utt2lang', 'w') as f:
        for line in file_list:
            line += '\n'
            f.write(line)
    return C

def Average(P):
    n  = len(P)
    m = len(P[0])
    t = [0]*(m-1)
    for j in range(m-1):
        l = []
        for i in range(n):  
            if P[i][-1] == j:
                l.append(P[i][j])
        t[j] = sum(l)/len(l)
    return t
C = count_conf_join('combine.txt')
