# Recommender System API

This is a Flask project containing the recommender system API for the colleges at Maastricht University.

## Installation

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the application using `python3 app.py`.

## Usage

1. Open a tool like Postman and create a POST request to `http://127.0.0.1:5000/api/recommend`.

The POST request needs to contain a JSON object like this as it's body:

{
"keywords": {
"dog": 0.4,
"cat": 0.8,
"animal": 0.9,
"pet": 0.7,
"cute": 0.6,
"adorable": 0.5,
"kitten": 0.8,
"puppy": 0.8
},
"bloom": {
"remember": 0.5,
"understand": 0.6,
"apply": 0.7,
"analyze": 0.8,
"evaluate": 0.9,
"create": 1.0
}
}

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push to your fork and submit a pull request.

## Credits

This project was created by [Your Name](https://github.com/gabeha).

## License

This project is licensed under the [MIT License](LICENSE).
