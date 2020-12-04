var mysql = require('mysql');
var express = require('express');
var session = require('express-session');
var bodyParser = require('body-parser');
var path = require('path');
var connection = mysql.createConnection({
	host     : 'localhost',
	user     : 'root',
	password : '3443',
	database : 'Hospital_Records'
});

var app = express();
app.use(express.static('../SE_UI'))
app.use(session({
	secret: 'secret',
	resave: true,
	saveUninitialized: true
}));
app.use(bodyParser.urlencoded({extended : true}));
app.use(bodyParser.json());
app.set('views','./Front/views');
app.set('view engine','pug');
app.get('/', function(request, response) {
	response.sendFile(path.join(__dirname + '/index.html'));
});
app.get('/process_post',function(request,response){
	//var username= request.body.username;

	//if(username)
	//{
     connection.query('SELECT * FROM paitient_details',function(err,rows,fields){
     	if (err) throw err
     	response.render('jerry',{ title:'Patient Details',items: rows})
     })
 //}
 //else
 //{
 	//response.send(username);
 	//response.send('Please correct your codes');
 //}
});
/*
app.post('/', function(request, response) {
	var username = request.body.username;
	var password = request.body.password;
	var result1;
	if (username && password) {
		connection.query('SELECT Disease FROM paitient_details WHERE name = ? AND Registration_Id = ?', [username, password], function(error, results, fields) {
			if (results.length > 0) {
				request.session.loggedin = true;
				request.session.username = username;
				request.session.result1=results;
				console.log(results);
				response.redirect('/home');
				
			} else {
				response.send('Incorrect Username and/or Password! Kindly register if you are visiting us for the first time');
			}			
			response.end();
		});
		
	} else {
		response.send('Please enter Username and Password!');
		response.end();
	}
});

app.get('/home', function(request, response) {
	if (request.session.loggedin) {
		response.send('Welcome back, ' + request.session.username + '!'+ 'You are Suffering from:');
	} else {
		response.send('Please login to view this page!');
	}
	response.end();
});
*/
app.listen(8080);