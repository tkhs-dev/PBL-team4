package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.domain.repos.ClientRepository
import io.ktor.server.application.*
import io.ktor.server.auth.*
import org.koin.ktor.ext.inject

fun Application.configureSecurity() {
    val clientRepo by inject<ClientRepository>()
    install(Authentication){
        bearer {
            authenticate { cred ->
                val user = clientRepo.findClientBySecret(cred.token)
                if (user != null) {
                    UserIdPrincipal(user.id.toString())
                } else {
                    null
                }
            }
        }
    }
}
