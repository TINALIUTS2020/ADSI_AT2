class HomePage(object):
    def __init__(self) -> None:
        pass

    @property
    def page(self):
        # basic webpage
        # <link rel="stylesheet" href="./style.css">
        # <link rel="icon" href="./favicon.ico" type="image/x-icon">
        return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>ADSI AT2 BEER API!</title>
  </head>
  <body>
    <main>
        <h1>ADSI AT2 BEER API!</h1>
        <p><button type="button" onclick="window.location.href='/docs';">Go To Docs</button></p>

        <p>
        info = (
        'description': 'This API provides predictions for beer types based on given input parameters.',
        'endpoints': (
            '/': 'Display project information and endpoints',
            '/health/': 'Return a welcome message',
            '/beer/type/': 'Return prediction for a single input',
            '/beers/type/': 'Return predictions for multiple inputs',
            '/model/architecture/': 'Display the architecture of the Neural Networks'
        ),
        'input_parameters': (
            'brewery_name': 'string',
            'review_aroma': 'float',
            'review_appearance': 'float',
            'review_palate': 'float',
            'review_taste': 'float',
            'beer_abv': 'float'
        ),
        'output_format': 'JSON',
        'github_repo': 'https://github.com/TINALIUTS2020/ADSI_AT2'
    )
    </p>
    </main>
	<script src="index.js"></script>
  </body>
</html>
"""