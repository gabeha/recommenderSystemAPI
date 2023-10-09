from rec_sys_uni.recommender_system import RecSys

rs = RecSys()

# Print settings of RecSys
rs._settings_()

# Change settings of RecSys
rs.top_n = 1

rs._settings_()

# Get list of recommended courses
student_input = { "keywords": { "python": 0.5, "data science": 0.2 },
                  "blooms": { "create": 0.5, "understand": 0.75, "apply": 0.25, "analyze": 0.5, "evaluate": 0.0, "remember": 1.0 } }

# Check source code of rec_sys_uni.recommender_system
results = rs.get_recommendation(student_input)

# Example of the output
print(results)

#%%

#%%
