import joblib
import json
import os
from tqdm.auto import tqdm
import pandas as pd


class WarningModel:
    def __init__(self):
        # Iterate through all the models in the file
        model_filename = 'rec_sys_uni/datasets/data/student/warning_models'
        # iterate over files in
        # that directory
        self.models = {}
        self.columns_order = ['COR1002', 'COR1003', 'COR1004', 'COR1006', 'HUM1003', 'HUM1007', 'HUM1010', 'HUM1011', 'HUM1012', 'HUM1013', 'HUM1016', 'HUM2003', 'HUM2005', 'HUM2007', 'HUM2008', 'HUM2013', 'HUM2016', 'HUM2018', 'HUM2021', 'HUM2022', 'HUM2030', 'HUM2031', 'HUM2046', 'HUM2047', 'HUM2051', 'HUM2054', 'HUM2056', 'HUM2057', 'HUM2058', 'HUM2059', 'HUM2060', 'HUM3014', 'HUM3019', 'HUM3029', 'HUM3034', 'HUM3036', 'HUM3040', 'HUM3042', 'HUM3043', 'HUM3044', 'HUM3045', 'HUM3049', 'HUM3050', 'HUM3051', 'HUM3052', 'HUM3053', 'SCI1004', 'SCI1005', 'SCI1009', 'SCI1010', 'SCI1016', 'SCI2002', 'SCI2009', 'SCI2010', 'SCI2011', 'SCI2017', 'SCI2018', 'SCI2019', 'SCI2022', 'SCI2031', 'SCI2033', 'SCI2034', 'SCI2035', 'SCI2036', 'SCI2037', 'SCI2039', 'SCI2040', 'SCI2041', 'SCI2042', 'SCI2043', 'SCI2044', 'SCI3003', 'SCI3005', 'SCI3006', 'SCI3007', 'SCI3046', 'SCI3049', 'SCI3050', 'SCI3051', 'SCI3052', 'SSC1005', 'SSC1007', 'SSC1025', 'SSC1027', 'SSC1029', 'SSC1030', 'SSC2002', 'SSC2004', 'SSC2006', 'SSC2007', 'SSC2008', 'SSC2009', 'SSC2010', 'SSC2011', 'SSC2018', 'SSC2019', 'SSC2020', 'SSC2022', 'SSC2024', 'SSC2025', 'SSC2027', 'SSC2028', 'SSC2029', 'SSC2037', 'SSC2039', 'SSC2043', 'SSC2046', 'SSC2048', 'SSC2050', 'SSC2053', 'SSC2055', 'SSC2060', 'SSC2061', 'SSC2062', 'SSC2063', 'SSC2064', 'SSC2065', 'SSC2070', 'SSC2071', 'SSC3002', 'SSC3003', 'SSC3006', 'SSC3008', 'SSC3009', 'SSC3011', 'SSC3012', 'SSC3013', 'SSC3017', 'SSC3018', 'SSC3019', 'SSC3023', 'SSC3030', 'SSC3032', 'SSC3033', 'SSC3034', 'SSC3036', 'SSC3038', 'SSC3040', 'SSC3041', 'SSC3047', 'SSC3049', 'SSC3051', 'SSC3052', 'SSC3054', 'SSC3055', 'SSC3056', 'SSC3059', 'SSC3060', 'SSC3061']
        print("Loading warning models: ")
        progress_bar = tqdm(range(len(os.listdir(model_filename))))
        for filename in os.listdir(model_filename):
            f = os.path.join(model_filename, filename)
            # Load the model
            progress_bar.update(1)
            self.models[filename[:7]] = joblib.load(f)
        # Opening JSON file
        with open('rec_sys_uni/datasets/data/student/warnings_rec_courses/rec_courses.json', 'r') as openfile:
            # Reading from json file
            self.rec_courses = json.load(openfile)

        print("Warning model loaded successfully!\n")



    def predict(self, student_info):
        X = pd.DataFrame(columns=self.columns_order)
        if student_info.student_data:
            for i in student_info.student_data['courses_taken']:
                X.loc[0, i] = student_info.student_data['courses_taken'][i]['grade']
            X.fillna(0, inplace=True)
        else:
            X.loc[0, :] = 0

        course_data = student_info.course_data
        for i in student_info.results["recommended_courses"]:
            if i in self.models:
                prediction = self.models[i].predict(X[self.models[i].feature_names_in_])
                if prediction[0] == 0:
                    student_info.results["recommended_courses"][i]["warning"] = True
                    recommended_courses = self.rec_courses[i]
                    for j in recommended_courses:
                        if student_info.student_data:
                            if j in student_info.student_data['courses_taken']:
                                continue
                        if recommended_courses[j]['cosine_similarity'] > 0.66:
                            json_object = {"course_code": j,
                                           "course_name": course_data[j]['course_name'],
                                           "match": round(recommended_courses[j]['cosine_similarity'] * 100, 1)}
                            student_info.results["recommended_courses"][i]["warning_recommendation"].append(json_object)
