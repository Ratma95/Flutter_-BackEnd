# Levantar el proyecto en ambiente local
Para ingresar al entorno virtual:
desde la ruta del proyecto lanzar el siguiente comando:

virtualenv .venv

para activar el entorno virtual, desde la ruta del proyecto lanzar el siguiente comando:
.venv\Scripts\activate

una vez activado el entorno virtual debe mostrar como se indica:
(.venv) PS C:\RutaDelProyecto

luego lanzar la instruccion para instalar las dependencias:
pip install -r requirements.txt

para desplegar en ambiente local:
python app.py runserver

# Endpoints:
_____________
> get Algorithms :  METHOD: GET http://127.0.0.1:5001/algorithms
_____________
> Metodo del Codo : METHOD POST http://127.0.0.1:5001/MetododelCodo

> clasificacion : METHOD POST http://127.0.0.1:5001/clasificacion

> dbscan : METHOD POST http://127.0.0.1:5001/dbscan

# QUERY: 

6 COLUMNS:
{
    "query":"select o.htitulo, f_dimPuestoEmpleo(o.id_oferta,7) as funciones, f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento, f_dimPuestoEmpleo(o.id_oferta,3) as competencias, f_dimPuestoEmpleo(o.id_oferta,2) as habilidades, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null limit 500;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 30
      }

14 COLUMNS:
{
    "query":"select o.htitulo_cat, o.htitulo, w.pagina_web, o.empresa, o.lugar, o.salario, date_part('year',o.fecha_publicacion) as periodo, f_dimPuestoEmpleo(o.id_oferta,7) as funciones, f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento, f_dimPuestoEmpleo(o.id_oferta,3) as habilidades, f_dimPuestoEmpleo(o.id_oferta,2) as competencias, f_dimPuestoEmpleo(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPuestoEmpleo(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null limit 500;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 30
      }

# QUERYS SIN LIMITES: 

6 COLUMNS:
{
    "query":"select o.htitulo, f_dimPuestoEmpleo(o.id_oferta,7) as funciones, f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento, f_dimPuestoEmpleo(o.id_oferta,3) as competencias, f_dimPuestoEmpleo(o.id_oferta,2) as habilidades, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 30
      }

14 COLUMNS:
{
    "query":"select o.htitulo_cat, o.htitulo, w.pagina_web, o.empresa, o.lugar, o.salario, date_part('year',o.fecha_publicacion) as periodo, f_dimPuestoEmpleo(o.id_oferta,7) as funciones, f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento, f_dimPuestoEmpleo(o.id_oferta,3) as habilidades, f_dimPuestoEmpleo(o.id_oferta,2) as competencias, f_dimPuestoEmpleo(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPuestoEmpleo(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 30
      }

