attr1 = float(retrievedFinancials['income_statement']['NetProfitMargin'])*\
        float(retrievedFinancials['income_statement']['Revenue']) / \
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])
        
attr3 = float(retrievedFinancials['company_key_metrics']['WorkingCapital'])/ \
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])
        
        
attr4 = float(retrievedFinancials['balance_sheet_statement']['Totalcurrentassets'])/ float(retrievedFinancials['balance_sheet_statement']['Totalcurrentliabilities'])
        
        
attr5 = 365*(float(retrievedFinancials['balance_sheet_statement']['Cashandcashequivalents'])+ \
        float(retrievedFinancials['balance_sheet_statement']['Cashandshort-terminvestments']) + \
        float(retrievedFinancials['balance_sheet_statement']['Receivables']) - \
        float(retrievedFinancials['balance_sheet_statement']['Totalcurrentliabilities']) / \
        float(retrievedFinancials['income_statement']['OperatingExpenses']) - \
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) ) 
        
        
attr8 = float(retrievedFinancials['company_key_metrics']['BookValueperShare'])* \
        float(retrievedFinancials['income_statement']['Revenue']) / \
        float(retrievedFinancials['company_key_metrics']['RevenueperShare']) - \
        float(retrievedFinancials['balance_sheet_statement']['Totalliabilities']) 
        
        
attr9 = float(retrievedFinancials['income_statement']['Revenue'])/ \
        float(retrievedFinancials['balance_sheet_statement']['Totalassets']) 
        
        
attr12 = float(retrievedFinancials['income_statement']['GrossProfit'])/ \
        float(retrievedFinancials['balance_sheet_statement']['Totalcurrentliabilities']) 
        
        
attr13 = float(retrievedFinancials['income_statement']['GrossProfit'])+ \
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) /\
        float(retrievedFinancials['income_statement']['Revenue'])        
        
        
attr15 = float(retrievedFinancials['balance_sheet_statement']['Totalliabilities'])*365 / \
        (float(retrievedFinancials['income_statement']['GrossProfit']) +\
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) )       
        
        
attr16 = float(retrievedFinancials['income_statement']['GrossProfit'])+ \
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalliabilities'])
        
        
attr19 = float(retrievedFinancials['income_statement']['GrossProfit'])/ \
        float(retrievedFinancials['income_statement']['Revenue'])         
        
        
attr20 = float(retrievedFinancials['balance_sheet_statement']['Inventories'])*365/ \
        float(retrievedFinancials['income_statement']['Revenue'])         
        

attr21 = float(retrievedFinancials['income_statement']['Revenue'])/ \
        float(retrievedFinancials['income_statement-1']['Revenue'])         
        
attr22 = float(retrievedFinancials['income_statement']['OperatingIncome'])- \
        float(retrievedFinancials['income_statement']['OperatingExpenses'])/\
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])               
        
        
attr24 = float(retrievedFinancials['income_statement']['GrossProfit'])/ \
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])               
        
        

attr27 = (float(retrievedFinancials['income_statement']['OperatingIncome'])-\
        float(retrievedFinancials['income_statement']['OperatingExpenses'])) / \
        float(retrievedFinancials['income_statement']['InterestExpense'] + 1)             
        
attr28 = float(retrievedFinancials['company_key_metrics']['WorkingCapital'])/ \
        float(retrievedFinancials['balance_sheet_statement']['Totalnon-currentassets'])               
        
        
attr30 = float(retrievedFinancials['balance_sheet_statement']['Totalliabilities'])- \
        float(retrievedFinancials['balance_sheet_statement']['Cashandcashequivalents']) /\
        float(retrievedFinancials['income_statement']['Revenue'])              
        
        
attr32 = float(retrievedFinancials['balance_sheet_statement']['Totalcurrentliabilities'])*365 /\
        float(retrievedFinancials['income_statement']['CostofRevenue'])       
        
        
attr36 = float(retrievedFinancials['income_statement']['Revenue'])/\
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])        
        
attr37 = float(retrievedFinancials['balance_sheet_statement']['Totalcurrentassets']) - \
        float(retrievedFinancials['balance_sheet_statement']['Inventories']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalnon-currentliabilities'])
        
        
attr41 = float(retrievedFinancials['balance_sheet_statement']['Totalliabilities']) / \
        float(retrievedFinancials['income_statement']['OperatingIncome']) - \
        float(retrievedFinancials['income_statement']['OperatingExpenses']) + \
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) * (12 /365)
        
        
attr47 = float(retrievedFinancials['balance_sheet_statement']['Inventories']) / \
        float(retrievedFinancials['income_statement']['CostofRevenue'])
        
        
attr50 = float(retrievedFinancials['balance_sheet_statement']['Totalcurrentassets']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalliabilities'])
        
        
attr51 = float(retrievedFinancials['balance_sheet_statement']['Totalcurrentliabilities']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalassets'])
        
        
attr53 = float(retrievedFinancials['balance_sheet_statement']['Totalshareholdersequity']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalnon-currentassets'])
        
        
attr59 = float(retrievedFinancials['balance_sheet_statement']['Totalnon-currentliabilities']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalshareholdersequity'])

        
attr64 = float(retrievedFinancials['income_statement']['Revenue']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalnon-currentassets'])
        
attr65 = float(retrievedFinancials['income_statement']['EBIT']) /\
        float(retrievedFinancials['income_statement']['InterestExpense'] + 1)
        
        
attr66 = float(retrievedFinancials['income_statement']['EBIT']) +\
        float(retrievedFinancials['cash_flow_statement']['Depreciation&Amortization']) /\
        float(retrievedFinancials['balance_sheet_statement']['Totalliabilities'])
        
print(attr1, attr3, attr4, attr5, attr8, attr9, attr12, attr13, attr15, attr16, attr19, attr20, attr21, attr22, attr24, attr27, attr28, attr30, attr32, attr36, attr37, attr41, attr47, attr50, attr51, attr53, attr59, attr64, attr65, attr66)
