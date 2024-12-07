package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.dao.Assignments
import ac.osaka_u.ics.pbl.dao.Clients
import ac.osaka_u.ics.pbl.dao.Models
import ac.osaka_u.ics.pbl.dao.Tasks
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.resources.*
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.transactions.transaction
import org.postgresql.ds.PGSimpleDataSource

fun main(args: Array<String>) {
    embeddedServer(Netty, port = 8080) {
        module()
    }.start(wait = true)
}

fun Application.module() {
    configureDatabase()
    configureSecurity()
    install(Resources)
    configureRouting()
    configureStatus()
    install(ContentNegotiation) {
        json()
    }
}
