def sentence_pairing(raw_words, tokenized_words):
    len_r = len(raw_words)
    len_t = len(tokenized_words)
    i = 0
    j = 0
    set_list = []
    r_cur = ""
    t_cur = ""
    pair = [[], []]
    while i < len_r and j < len_t:
        flag = False
        if r_cur == "" and t_cur == "":
            r_cur = raw_words[i]
            t_cur = tokenized_words[j]
            pair = [[i], [j]]
        if len(r_cur) == len(t_cur):
            flag = True
        elif len(r_cur) > len(t_cur):
            j += 1
            t_cur += tokenized_words[j]
            pair[1].append(j)
        elif len(r_cur) < len(t_cur):
            i += 1
            r_cur += " " + raw_words[i]
            pair[0].append(i)
        if flag:
            i += 1
            j += 1
            set_list.append(pair)
            r_cur = ""
            t_cur = ""
            pair = [[], []]
    return set_list

def pair_list_to_dict(set_list):
    dic = {}
    for pair in set_list:
        for idx in pair[1]:
            dic[idx] = pair[0]
    return dic

if __name__ == '__main__':
    sent = "tom & jerry leave there at 11:00."
    r_words = ['tom', '&', 'jerry', 'leave', 'there', 'at', '11:00', '.', '.', '.', 'asd']
    t_words = ['tom & jerry', 'leave', 'there', 'at', '11', ':', '00', '.', '.', '. asd']
    set_list = sentence_pairing(['--', 'Dimitris', 'Kontogiannis', ',', 'Athens', 'Newsroom', '+301', '3311812-4'],
                                ['--', 'Dimitris', 'Kontogiannis', ',', 'Athens', 'Newsroom', '+301 3311812', '-4'])
    print(pair_list_to_dict(set_list))
    # print(pair_list_to_dict(set_list))