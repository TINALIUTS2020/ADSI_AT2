docker compose build api_prod --no-cache
cd conntainers/api_prod
docker tag adsi_at2-api_prod registry.heroku.com/adsi-attwo-utsthree/web
docker push registry.heroku.com/adsi-attwo-utsthree/web
heroku container:release web --app adsi-attwo-utsthree