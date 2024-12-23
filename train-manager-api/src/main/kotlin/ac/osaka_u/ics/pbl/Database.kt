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
        val url = environment.config.propertyOrNull("ktor.deployment.database.url")?.getString() ?: "localhost"
        val port = environment.config.propertyOrNull("ktor.deployment.database.port")?.getString() ?: "5432"
        user = environment.config.propertyOrNull("ktor.deployment.database.user")?.getString() ?: "postgres"
        password = environment.config.propertyOrNull("ktor.deployment.database.password")?.getString() ?: "postgres"
        setURL("jdbc:postgresql://${url}:${port}/train_manager")
    })
    transaction {
        log.info("DB CONNECTION SUCCESSFUL")
        SchemaUtils.createMissingTablesAndColumns(Clients, Tasks, Models, Assignments, TaskGenerators, Errors)
    }
}