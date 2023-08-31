CONTINUOUS_COLUMNS = ["MEMBERSHIP_TERM_YEARS", "ANNUAL_FEES", "MEMBER_ANNUAL_INCOME", "MEMBER_OCCUPATION_CD",
                      "MEMBER_AGE_AT_ISSUE", "ADDITIONAL_MEMBERS", "AGENT_PERFORMANCE", "INCOME_TO_FEE_RATIO",
                      "START_YEAR", "START_MONTH", "START_DAY", "START_YEAR_AGE_COMBINED", "DAYS_STAY"
                     ]
CATEGORICAL_COLUMNS = ["MEMBER_MARITAL_STATUS", "MEMBER_GENDER", "MEMBERSHIP_PACKAGE", "PAYMENT_MODE"]

DROP_COLUMNS_TRAIN = ['ID', 'MEMBERSHIP_NUMBER', 'START_DATE', 'END_DATE',
                      'MEMBERSHIP_STATUS','AGENT_CODE'
               ]

DROP_COLUMNS_TEST = ['ID', 'MEMBERSHIP_NUMBER', 'START_DATE', 'AGENT_CODE',
                     'MEMBERSHIP_STATUS']