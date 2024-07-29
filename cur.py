import py_vncorenlp
import math

inp = open('mim_heading.txt', 'r', encoding='utf-8')
article_word = open('mim_heading_count.txt', 'r', encoding='utf-8') #tổng số từ trong article 
fileout = open(file='ans.txt', mode='w+', encoding='utf-8') 
alldiff = open('alldiff.txt', 'r', encoding='utf-8')

score = {}
idf_word = {}
article_num = 2816 #number of articles in data
diffword = alldiff.readlines()

for word in diffword:
    if word.endswith('\n'): word = word[0:-1]
    word = word.split(' ')
    word[1] = int(word[1])
    
for word in diffword:
    if word.endswith('\n'): word = word[0:-1]
    word = word.split(' ')
    word[1] = int(word[1])
    score[word[0]] = article_num / word[1]
    idf_word[word[0]] = math.log10(score[word[0]])

model = py_vncorenlp.VnCoreNLP(save_dir='/Users/DeLL/Documents/Code/Current')

questions = ['đĩa đệm']

available = ['N', 'V', 'Np', 'P', 'A']

cnt = {}

key_count = 0

for head in questions:
    head = head.lower()
    head = head.replace('huyết áp cao', 'tăng huyết áp')
    head = head.replace('huyết áp thấp', 'giảm huyết áp')
    ans = ''
    alltext = model.annotate_text(head)
    model.print_out(alltext)
    for sentence in alltext:
        for ele in alltext[sentence]:
            if ele['posTag'] not in available: continue
            if ele['wordForm'] == 'tài_liệu' or ele['wordForm'] == 'tham_khảo': continue
            if ele['wordForm'] == 'thông_tin' or ele['wordForm'] == 'thêm': continue
            if ele['wordForm'] == 'sự': continue
            if ele['wordForm'] == 'chung': continue
            if ele['wordForm'] == 'điểm': continue
            if ele['wordForm'] == 'chính': continue
            if ele['wordForm'] == 'bị' and ele['posTag'] == 'V': continue
            if ele['wordForm'] == 'đái_tháo_đường':
                ele['wordForm'] = 'tiểu_đường'
            if ele['wordForm'] in cnt.keys():
                cnt[ele['wordForm']] += 1   
            else: cnt[ele['wordForm']] = 1
    print(cnt.keys().__len__(), file=fileout, flush=True)
    for item in cnt:
        print(item, file=fileout, flush=True)
        print(cnt[item], file=fileout, flush=True)

for item in cnt:
    key_count += 1

all = []

while inp.readable():
    link = inp.readline()[0:-1]
    if link == '': break
    point = 0
    key_num = 0
    article_word_num = int(article_word.readline()[0:-1])
    num = int(inp.readline()[0:-1])

    for i in range(num):
        cur_word = inp.readline()[0:-1]
        cur_num = int(inp.readline()[0:-1])
        tf_word = cur_num / article_word_num
        if cur_word in cnt.keys():
            point += cnt[cur_word] * (tf_word * idf_word.get(cur_word, 0)) #số lần key xuất hiện trong search * tf-idf
            key_num += 1
        
    point *= (key_num * key_num) / key_count #tỷ lệ xuất hiện keyword

    all.append((point, link))

def cmp(val):
    return val[0]
all.sort(key=cmp, reverse=True)

print(all, file=fileout)