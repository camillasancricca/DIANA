import efficient_apriori


def rules(store_data, support, confidence, actual_rules, checks, violations):
    f = open("../Risultati/FD_eff_a_priori.txt", 'a')
    # every tuple is a record, the index of the column is also stored
    records = []
    for i in range(0, len(store_data)):
        records.append([(str(store_data.values[i, j]), j) for j in range(0, len(store_data.columns))])

    # find the rules using apriori algorithm
    itemset, rules = efficient_apriori.apriori(records, min_support=support, min_confidence=confidence, verbosity=1)

    #if actual_rules is not None:
    #    for rule in rules:
    #        checks += 1
    #        if rule not in actual_rules:
    #            violations += 1
    #            rules.remove(rule)


    if actual_rules is not None:
        for rule in actual_rules:
            checks += 1
            if rule not in rules:
                violations += 1
                actual_rules.remove(rule)
    else:
        actual_rules = rules

    #print(rules,'\n')
    f.write("\n")
    f.write(str(actual_rules))
    f.close()

    return actual_rules, checks, violations
