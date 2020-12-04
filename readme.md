1)For testing this software folllowing are the requirements :
   Software Requirements:
   i)   Visual Studio Code (VSC) for Python.

   ii)  Python 3.8 with Tkinter and Anaconda libraries installed.

   iii) Nodejs and Expressjs.

   iv)  MySQL 5.7 

   v)   MySQL connecteor for Python 

   Installing python libraries:

   i)Tkinter: go to Activestate.com, and follow along the links to download the Community Edition of ActivePython for Windows. Make sure you're downloading a 3.1 or newer version, not a 2.x version.Run the installer, and follow along. You'll end up with a fresh install of ActivePython.

   ii)Pandas:use the command "pip install pandas" on the command prompt.

   iii)sklearn:use the command "pip install -U scikit-learn" on the command prompt.

   iv)Pillow : use the command "pip install Pillow" on the command prompt. 

   v)Matplotlib : use the command "pip install matplotlib" on the command prompt.

   vi)MySQL connector : use he command "pip install mysql-connector-python" on the command prompt.

 2) Steps involved in testing the software are as follows:

    Files named "Training.csv" and "Testing.csv" are the dataset files and the "hospital_records_paitient_details.sql" is a dump file consisting of queries necessary to recreate the orignal database.
    The master directory consists of "SE_UI directory" that contains Front-end and Back-end files for the patient's website.
    Download the entire master directory named "Pro_Codes_Datasets_and_UI_files". 
    
    i) Open MySQL terminal and create a database named "Hospital_Records" using the command "CREATE DATABASE Hospital_Records;" and then open the dump file named "hospital_records_patient_details.sql" in the project directory and execute all the queries given in the dump on the MySQL terminal to create all the tables with the required entries and keys.

    ii)Open the command prompt and go the path of the project directory SE_UI.

    iii)Type "node log.js" to run the server on the localhost 8080.

    iv)Now open  index.html file in the SE_UI directory is the login page. If you are a new user then you would get an error upon login and would be prompted for registration.

    v)Open another command prompt and go the path of the SE_UI dorectory and type "cd Front". Then type "node backend.js" to run the server on port 8000 and then open  the file named index.html and fill in the desired details in the registration form and click on submit button.

    vi) Now open the MySQL terminal and check for the database named "Hospital_Records" by using the command "use Hospital_records" and then type "SELECT * FROM PAITIENT_DETAILS" to check the info about the patients in the database. The Disease column is left NULL for the new patient.

    vii) 
        a) Open another command prompt and copy the path of the algos.py file to the the main path and then run the command "python algos.py" upon which   a gui window will pop-up . 

        b) Fill in the Registration_Id you entered in the form or refer the Registration_Id column in the database and type the value in the Registration_Id column in the text box . 

        c) Fill in all the symptoms and click on the button of your choice indicating the algorithm  with which you want to predict the disease.It's desirable that you click on all the three buttons. The algos.py file is only visible to the doctor.

    viii) Click on the finalize button after the diseases are displayed in the text-box.

    ix)Display the table again in the MySQL terminal and you will find that the disease-column which was previously NULL has been updated to one of the diseases predicted by either "Decision Tress","RandomForests" or "Naive Bayes".

    x) Open the previous terminal running on port 8080 and press "ctrl + 5 + c" and restart the server by running the command "node log.js".

    xi)Open the index.html in the SE_UI directory and enter your name in the username as given in the database under the column of "name" and enter Registration_Id as your password and click on Login .

    xii)Check the terminal running the running the server on port 8080 to check the disease the patient is suffering.