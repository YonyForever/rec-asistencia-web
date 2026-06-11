from database.connection import init_db, get_session
from database.models import ClaseConfig

def main():
    print("Inicializando Base de Datos local...")
    init_db()
    print("Base de datos e infraestructura de tablas inicializada correctamente.")
    
    # Insertar la clase real si la base de datos está vacía
    session = get_session()
    try:
        if session.query(ClaseConfig).count() == 0:
            print("Creando configuración de clase real...")
            clase_real = ClaseConfig(
                nombre_materia="Formulaciòn y Gestiòn de Proyectos",
                hora_inicio="13:00:00",
                limite_presente=5,   # 0-5 min -> PRESENTE
                limite_tarde=20,    # 6-20 min -> TARDE
                activo=1            # Activo
            )
            session.add(clase_real)
            session.commit()
            print("Clase 'Formulaciòn y Gestiòn de Proyectos' (13:00:00 PM) agregada con éxito.")
    except Exception as e:
        session.rollback()
        print(f"Error al insertar configuración inicial: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    main()
