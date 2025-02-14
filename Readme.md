# **Data Analysis Web Application**

## **1. Introduction**

The project involves developing a web application based on Django that enables data analysis and prediction using machine learning models. The objective is to provide an intuitive platform that allows users to import datasets, perform statistical analyses, and visualize results through interactive charts.

## **2. Project Objectives**

- Provide a simple and user-friendly web interface for data analysis.
- Allow the import of CSV and Excel files.
- Integrate machine learning models for predictions.
- Generate interactive visualizations to help interpret results.

## **3. Technologies Used**

- **Back-end**: Django, Django REST Framework
- **Front-end**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite
- **Machine Learning Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## **4. Main Features**

- **Data Import**: Supports CSV and Excel formats.
- **Data Exploration**: Computes descriptive statistics and displays them in tables.
- **Visualization**: Charts for data analysis.
- **Prediction**: Uses machine learning models to make predictions based on imported data.

## **5. System Architecture**

The project follows an MVC (Model-View-Controller) architecture:

- **Model**: Defines data structures and relationships.
- **View**: Manages user interface and interactions.
- **Controller**: Handles requests, applies logic, and returns results.

## **6. Results and Analysis**

The application meets the initial requirements and allows efficient data exploration. The machine learning models provide accurate predictions, and the user interface is intuitive with optimized performance.

## **7. Future Improvements**

- Add a REST API to enable integration with other systems.
- Extend supported file formats for data import.
- Improve prediction models using advanced algorithms.
- Implement an automated reporting system.

## **8. Conclusion**

This project successfully delivers a high-performance web application for data analysis and prediction. It provides an intuitive solution for data analysts and researchers looking to explore and visualize datasets without requiring extensive programming knowledge.

---

## **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/NizarBelaatik/Data_Analysis_Web_App
   cd django-project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply database migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the server**
   ```bash
   python manage.py runserver
   ```

The application will be accessible at `http://127.0.0.1:8000/`.
