import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load column names
with open("column name.txt", "r") as f:
    name = f.readlines()
name = [i.strip() for i in name]

inputs = {}
st.title("Prediction-Churn Dataset")
for i, j in enumerate(name):
    if i!=1 and i!=6 and i!=7:
        inputs[j] = st.text_input(label=j)
    elif i == 1:
        inputs[j]=st.selectbox(label=j,options=["Male","Female"])
    elif i == 6:
        inputs[j]=st.selectbox(label=j,options=["Premium","Standard","Basic"])
    elif i == 7:
        inputs[j]=st.selectbox(label=j, options=["Monthly","Quarterly","Annual"])
if st.button("OnSubmit"):
    # Load model and preprocessing files
    model = load_model("model.h5")
    
    with open("LabelEncoder_c.pkl", "rb") as f:
        lb_c = pickle.load(f)
    with open("LabelEncoder_g.pkl", "rb") as f:
        lb_g = pickle.load(f)
    with open("LabelEncoder_s.pkl", "rb") as f:
        lb_s = pickle.load(f)
    with open("Minmaxsacler.pkl", "rb") as f:
        Ms = pickle.load(f)
    
    # Gender encoding
    gender_input = inputs.get("Gender")
    subscription=inputs.get("Subscription Type")
    contract=inputs.get("Contract Length")
    if gender_input and subscription and contract:
        try:
            gender_encoded = lb_g.transform([gender_input])
            subscription_encode=lb_s.transform([subscription])
            contract_encode=lb_c.transform([contract])
            inputs["Subscription Type"]=subscription_encode[0]
            inputs["Contract Length"]=contract_encode[0]
            inputs["Gender"] = gender_encoded[0]
            st.write(f"Encoded Gender: {inputs['Gender']}{inputs['Contract Length']}{inputs["Subscription Type"
                                                                                            ]}")
        except Exception as e:
            st.error(f"Error encoding gender: {e}")
    
    try:
        input_values = [inputs[col] for col in name]

        input_values_scaled = Ms.transform([input_values])
        input_values_scaled = np.delete(input_values_scaled, -1) 
        input_values_scaled = np.expand_dims(input_values_scaled, axis=0) # Apply scaling
        
        # Predict
        prediction = model.predict(input_values_scaled)
        if prediction[0]==1:
            st.write("**Yes**")
            st.balloons()
        else:
            st.write("**No**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

   
   
            


