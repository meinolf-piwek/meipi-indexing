"""Liefert PostgreSQL database connection engine und definiert DB-Operationen."""

from logging import Logger, INFO
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, insert
from .model import Base as ModelBase


# Create the PostgreSQL engine and Sessionmaker
class pgEngine:
    def __init__(
        self,
        db_conn_string,
        metadata=ModelBase.metadata,
        logger: Logger = Logger("sqlalchemy.engine", level=INFO),
        enginekwargs: dict = None,
        sessionkwargs: dict = None,
    ):
        enginekwargs = {} if not enginekwargs else enginekwargs
        sessionkwargs = {} if not sessionkwargs else sessionkwargs
        self.logger = logger
        self.metadata = metadata
        try:
            self.engine = create_engine(db_conn_string, **enginekwargs)
            self.Session = sessionmaker(
                bind=self.engine, expire_on_commit=False, **sessionkwargs
            )
        except Exception as e:
            self.logger.error("Error creating PostgreSQL engine: %s", e)
            raise
        else:
            self.logger.info("PostgreSQL engine created successfully.")

    def recreate_tables(self):
        """Recreate tables in the database."""
        self.metadata.drop_all(self.engine)
        self.metadata.create_all(self.engine)

    def get_session(self, **kwargs)-> Session:
        return self.Session(**kwargs)

    def bulk_insert(self, TableClass: ModelBase, data: list[dict]):
        """Insert a list of data into the database."""
        with self.get_session(expire_on_commit=False) as session:
            try:
                stmt = insert(TableClass).values(data)
                session.execute(stmt)
            except Exception as e:
                self.logger.error("Error %s inserting", e)
                session.rollback()
                return False
            else:
                session.commit()
                self.logger.warning(
                    "Inserted data into %s table", TableClass.__tablename__
                )
                return True
