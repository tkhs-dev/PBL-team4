package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.domain.repos.*
import ac.osaka_u.ics.pbl.handler.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.resources.*
import org.koin.core.module.dsl.bind
import org.koin.core.module.dsl.singleOf
import org.koin.dsl.module
import org.koin.ktor.plugin.Koin
import org.koin.logger.slf4jLogger

fun main(args: Array<String>): Unit = EngineMain.main(args)

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
    onDebug {
//        val clientRepo = get<ClientRepository>()
//        clientRepo.createClient(Client(-1, "test"))
    }
}

fun Application.configureDI() {
    install(Koin) {
        slf4jLogger()
        modules(module {
            singleOf(::TaskRepositoryImpl){bind<TaskRepository>()}
            singleOf(::TaskGeneratorRepositoryImpl){bind<TaskGeneratorRepository>()}
            singleOf(::AssignmentRepositoryImpl){bind<AssignmentRepository>()}
            singleOf(::QueueRepositoryImpl){bind<QueueRepository>()}
            single{ModelRepositoryImpl(this@configureDI.staticRoot) as ModelRepository}
            singleOf(::ErrorRepositoryImpl){bind<ErrorRepository>()}
            singleOf(::ClientRepositoryImpl){bind<ClientRepository>()}
            singleOf(::TasksHandler)
            singleOf(::AssignmentsHandler)
            singleOf(::QueueHandler)
            singleOf(::ClientsHandler)
        })
    }
}

fun Application.onDebug(block: Application.() -> Unit) {
    if (debug) block()
}

val Application.staticRoot: String
    get() = environment.config.propertyOrNull("ktor.deployment.staticRoot")?.getString() ?: "./static"

val Application.debug: Boolean
    get() = environment.config.propertyOrNull("ktor.deployment.debug")?.getString()?.toBoolean() ?: false
