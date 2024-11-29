package ac.osaka_u.ics.pbl

import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.resources.*

fun main(args: Array<String>) {
    embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) {
            json()
        }
        install(Resources)
    }.start(wait = true)
}

fun Application.module() {
    configureSecurity()
    configureRouting()
}
