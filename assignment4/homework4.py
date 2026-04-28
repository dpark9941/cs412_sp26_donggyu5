import itertools


def apriori(transaction_filename, cost_filename, s, k, m):
    transactions = []
    with open(transaction_filename, "r") as f:
        for line in f:
            if line.strip():
                transactions.append(set(line.strip()))
                
    N = len(transactions)
    
    costs = {}
    with open(cost_filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                costs[parts[0]] = int(parts[1])

    result = {}
    if N == 0:
        return result

    def get_sum_price(itemset):
        return sum(costs[item] for item in itemset)
        
    def get_min_price(itemset):
        return min(costs[item] for item in itemset)

    C1_counts = {}
    for t in transactions:
        for item in t:
            fs = frozenset([item])
            C1_counts[fs] = C1_counts.get(fs, 0) + 1

    C = C1_counts
    F = {}
    L = 1

    while C:
        result[L] = {"c": {}, "f": {}}
        F_current = {}
        
        for c_itemset, count in C.items():
            c_str = "".join(sorted(list(c_itemset)))
            result[L]["c"][c_str] = count
            
            sum_price = get_sum_price(c_itemset)
            min_price = get_min_price(c_itemset)
            
            if count >= s and sum_price <= k and min_price >= m:
                result[L]["f"][c_str] = count
                F_current[c_itemset] = count
                
        F = F_current
        
        if not F:
            break
            
        L += 1
        C = {}
        items_list = list(F.keys())
        
        for i in range(len(items_list)):
            for j in range(i+1, len(items_list)):
                c1 = sorted(list(items_list[i]))
                c2 = sorted(list(items_list[j]))
                
                if c1[:-1] == c2[:-1]:
                    new_itemset = frozenset(c1).union(frozenset(c2))
    
                    valid = True
                    for item in new_itemset:
                        subset = new_itemset - frozenset([item])
                        if subset not in F:
                            valid = False
                            break
                            
                    if valid and new_itemset not in C:
                        count = sum(1 for t in transactions if new_itemset.issubset(t))
                        C[new_itemset] = count

    keys_to_delete = [level for level, data in result.items() if not data["c"] and not data["f"]]
    for key in keys_to_delete:
        del result[key]

    return result

def mine_rules(transaction_filename, cost_filename, s, k, m, min_conf):
    """
    Returns a dictionary mapping rule strings to metrics.
    Rule strings are 'A->BC' with each side alphabetically sorted.
    Only rules with confidence >= min_conf are returned.
    """
    apriori_res = apriori(transaction_filename, cost_filename, s, k, m)
    
    transactions = []
    with open(transaction_filename, "r") as f:
        for line in f:
            if line.strip():
                transactions.append(set(line.strip()))
    
    N = len(transactions)
    if N == 0:
        return {}

    frequent_dict = {}
    valid_frequent = []
    for level, data in apriori_res.items():
        for fs_str, count in data["f"].items():
            frequent_dict[fs_str] = count
            if level >= 2:
                valid_frequent.append((frozenset(fs_str), count))

    rules = {}
    for itemset, count in valid_frequent:
        for l_sub in range(1, len(itemset)):
            for left in itertools.combinations(itemset, l_sub):
                left_set = frozenset(left)
                right_set = itemset - left_set
                
                left_str = "".join(sorted(list(left_set)))
                right_str = "".join(sorted(list(right_set)))
                
                if left_str in frequent_dict and right_str in frequent_dict:
                    left_count = frequent_dict[left_str]
                    right_count = frequent_dict[right_str]
                    
                    conf = count / left_count
                    if conf >= min_conf - 1e-9:
                        sup = count / N
                        lift = conf / (right_count / N)
                        jacc_denom = (left_count / N) + (right_count / N) - sup
                        jaccard = sup / jacc_denom
                        
                        rule_str = left_str + "->" + right_str
                        rules[rule_str] = {
                            "support_count": count,
                            "support": round(sup, 6),
                            "confidence": round(conf, 6),
                            "lift": round(lift, 6),
                            "jaccard": round(jaccard, 6)
                        }
                    
    return rules
