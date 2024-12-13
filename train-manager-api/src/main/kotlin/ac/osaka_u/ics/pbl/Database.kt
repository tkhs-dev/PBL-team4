package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.data.dao.*
import io.ktor.server.application.*
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.transactions.transaction
import org.postgresql.ds.PGSimpleDataSource

fun Application.configureDatabase(){
    log.info("CONNECTING TO DB")
    Database.connect(PGSimpleDataSource().apply {
        setURL("jdbc:postgresql://localhost:5432/train_manager")
        user = "postgres"
        password = "postgres"
    })
    transaction {
        log.info("DB CONNECTION SUCCESSFUL")
        SchemaUtils.createMissingTablesAndColumns(Clients, Tasks, Models, Assignments, TaskGenerators)
    }
}