import psycopg2
from decouple import config 
con = psycopg2.connect(
  database=config('POSTGRESQL_DATABASE'), 
  user=config('POSTGRESQL_USER'), 
  password=config('POSTGRESQL_PASSWORD'), 
  host=config('POSTGRESQL_HOST'), 
  port=config('POSTGRESQL_PORT_BD'))