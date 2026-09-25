from mrz.checker.td3 import TD3CodeChecker
from datetime import datetime

def decode_mrz(mrz_extracted_text):
    #output of extract_mrz

    checker = TD3CodeChecker(mrz_text)

    if bool(checker):
        fields = checker.fields()
        # print(fields)
        mrz_data = {
            "document_type": fields.document_type,
            "document_number": fields.document_number,
            "full_name": fields.name + " " + fields.surname,
            "sex": fields.sex,
            "nationality": fields.nationality,
            "birth_date": datetime.strptime(fields.birth_date, "%y%m%d"),
            "document_expiry_date": datetime.strptime(fields.expiry_date, "%y%m%d"),
        }
    else:
        print("X MRZ failed validation checks.")