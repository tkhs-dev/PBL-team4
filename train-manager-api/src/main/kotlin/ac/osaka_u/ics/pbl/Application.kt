package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.domain.repos.*
import ac.osaka_u.ics.pbl.handler.TasksHandler
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.resources.*
import org.koin.core.module.dsl.bind
import org.koin.core.module.dsl.singleOf
import org.koin.dsl.module
import org.koin.ktor.plugin.Koin
import org.koin.logger.slf4jLogger

fun main(args: Array<String>) {
    embeddedServer(Netty, port = 8080) {
        module()
    }.start(wait = true)
}

fun Application.module() {
    configureDI()
    configureDatabase()
    configureSecurity()
    install(Resources)
    configureRouting()
    configureStatus()
    install(ContentNegotiation) {
        json()
    }
}

fun Application.configureDI() {
    install(Koin) {
        slf4jLogger()
        modules(module)
    }
}

val module = module {
    singleOf(::TaskRepositoryImpl){bind<TaskRepository>()}
    singleOf(::TaskGeneratorRepositoryImpl){bind<TaskGeneratorRepository>()}
    singleOf(::ClientRepositoryImpl){bind<ClientRepository>()}
    singleOf(::TasksHandler)
}
