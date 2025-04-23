import numpy as np
import pickle
import json
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from g4f.client import Client
def format_as_html_list(text: str) -> str:
    lines = text.strip().split("\n")
    html = "<ol>"
    in_sublist = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit() and '.' in line:
            if in_sublist:
                html += "</ul></li>"
            html += f"<li><strong>{line}</strong><ul>"
            in_sublist = True

        elif line.startswith("-") or line.startswith("–"):
            html += f"<li>{line[1:].strip()}</li>"

    if in_sublist:
        html += "</ul></li>"
    html += "</ol>"
    return html




# Load the machine learning model
model = pickle.load(open('best_model.pkl', 'rb'))

# Create application
app = Flask(__name__)

# JSON file to store user data
USER_DATA_FILE = 'users.json'

# Helper functions to read and write to JSON database
def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/loginpost', methods=['POST', 'GET'])
def userloginpost():
    global data1
    if request.method == 'POST':
        data1 = request.form.get('uname')
        data2 = request.form.get('password')

        print("Username:", data1)  # Debug statement
        print("Password:", data2)  # Debug statement

        if data2 is None:
            return render_template('login.html', msg='Password not provided')

        user_data = load_user_data()

        if data1 in user_data and user_data[data1]['password'] == data2:
            return render_template('index1.html')
        else:
            return render_template('login.html', msg='Invalid username or password')

@app.route('/NewUser')
def newuser():
    return render_template('NewUser2.html')

@app.route('/reg', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        uname = request.form.get('uname')
        email = request.form.get('email')
        phone = request.form.get('phone')
        age = request.form.get('age')
        password = request.form.get('psw')
        gender = request.form.get('gender')

        user_data = load_user_data()

        if uname in user_data:
            return render_template('NewUser2.html', msg='Username already exists')

        user_data[uname] = {
            'name': name,
            'email': email,
            'phone': phone,
            'age': age,
            'password': password,
            'gender': gender
        }

        save_user_data(user_data)
        return render_template('login.html')
    else:
        return render_template('NewUser2.html')

# Bind predict function to URL
@app.route('/showrf')
def showrf():
    return render_template('predict.html')

@app.route('/predictrf', methods=['POST'])
def predictrf():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)

    x = model.predict_proba(array_features)
    pos = x[0][1]
    pos = pos * 100

    # Prepare data for pie chart
    values = [('Positive', pos), ('Negative', 100 - pos)]

    # Generate pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([x[1] for x in values], labels=[x[0] for x in values], autopct='%1.1f%%')
    plt.title('Input Details')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static/input_pie_chart.png')  # Save the pie chart as a static file

    # Check the output values and retrieve the result with html tag based on the value
    if pos > 70:
        result_text = 'Probability of having heart disease: '
        risk_text = 'Risk is HIGH'
    elif pos > 40:
        result_text = 'Probability of having heart disease: '
        risk_text = 'Risk is MEDIUM'
    else:
        result_text = 'Probability of having heart disease: '
        risk_text = 'Risk is LOW'

    # Get detailed suggestion from gpt-4o-mini
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{
                        "role": "user",
                        "content": f"""
                    You are a medical assistant. A patient has a {pos}% probability of having heart disease. Based on this, provide a response ONLY in the following bullet-point format, and DO NOT use paragraphs or long descriptions:

                    1. **Risk Level**
                    - Risk: {pos}%
                    - Category: {"Low" if pos < 20 else "Moderate" if pos < 50 else "High" if pos < 80 else "Very High"}
                    - Immediate action: {"Monitor and maintain a healthy lifestyle." if pos < 50 else "Consult a healthcare provider for regular checkups."}
                    - Consider regular screenings for heart health if in the high-risk category.

                    2. **Dietary Recommendations**
                    - Eat more vegetables, fruits, and whole grains.
                    - Use olive oil or avocado instead of butter.
                    - Avoid sugary drinks and processed foods.
                    - Limit red meat and high-sodium meals to reduce cholesterol.

                    3. **Exercise Guidelines**
                    - Do 30 minutes of brisk walking, 5 days/week.
                    - Include light strength training twice a week to improve muscle strength.
                    - Stretch daily to improve flexibility and reduce muscle tension.
                    - Consider activities like yoga or swimming for overall cardiovascular health.

                    4. **Medication Advice**
                    {"- Monitor regularly; medication not needed." if pos < 20 else
                        "- Consider statins; check cholesterol and blood pressure regularly." if pos < 50 else
                        "- Statins and antihypertensives may be needed; follow doctor’s plan." if pos < 80 else
                        "- Statins, blood pressure medication, possibly aspirin. Frequent cardiologist visits required."}
                    - Follow prescribed medication schedules strictly.
                    - Keep track of side effects and report them to your doctor.
                    - Ensure regular checkups for monitoring medication effectiveness.

                    5. **Lifestyle Changes**
                    - Quit smoking completely to reduce the risk of heart disease.
                    - Limit or eliminate alcohol to maintain cardiovascular health.
                    - Get 7–9 hours of sleep each night to allow your heart to rest.
                    - Reduce stress with breathing exercises, mindfulness, or meditation.

                    6. **Medical Follow-Up**
                    - Book an appointment with a cardiologist for a thorough evaluation.
                    - Get an ECG (Electrocardiogram) to assess heart function.
                    - Consider a lipid profile and blood sugar tests to check for other risk factors.
                    - Repeat tests every 6–12 months to monitor changes in your heart health.

                    7. **Weight Management**
                    - Maintain a healthy weight to reduce strain on the heart.
                    - Incorporate both aerobic and strength exercises for weight loss.
                    - Track your weight and make necessary dietary adjustments.
                    - Consult a nutritionist for personalized meal planning.

                    8. **Blood Pressure Control**
                    - Regularly monitor your blood pressure.
                    - Reduce sodium intake to help control blood pressure levels.
                    - Engage in cardiovascular exercises to improve circulation.
                    - Take prescribed antihypertensive medication if necessary to manage high blood pressure.

                    9. **Hydration**
                    - Drink plenty of water throughout the day to keep your heart healthy.
                    - Avoid sugary drinks like soda that can contribute to heart disease risk.
                    - Opt for water-rich foods like cucumbers, melons, and berries.
                    - Limit alcohol consumption as it can dehydrate the body and strain the heart.

                    10. **Mental Health and Heart Health**
                    - Chronic stress can impact heart health, so manage stress effectively.
                    - Practice relaxation techniques such as deep breathing and meditation.
                    - Ensure a good work-life balance to avoid burnout and stress.
                    - Seek professional help if experiencing anxiety or depression, as mental health affects cardiovascular health.

                    **IMPORTANT**: Do not return long paragraphs. Follow this exact format.
                    """
}],




        web_search=False
    )
    detailed_suggestion = response.choices[0].message.content
    formatted_html = format_as_html_list(detailed_suggestion)

    return render_template('predict1.html', result=result_text, positive=pos, res2=risk_text, plot_image='static/input_pie_chart.png',detailed_suggestion=formatted_html)

@app.route('/model')
def show_model():
    return render_template('Model.html')

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, port=9860)
