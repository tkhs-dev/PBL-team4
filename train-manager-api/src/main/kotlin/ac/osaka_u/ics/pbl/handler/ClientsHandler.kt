package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.domain.model.Client
import ac.osaka_u.ics.pbl.domain.repos.ClientRepository
import ac.osaka_u.ics.pbl.model.ClientRequest
import ac.osaka_u.ics.pbl.model.ClientResponse
import ac.osaka_u.ics.pbl.model.toResponse
import java.security.SecureRandom
import java.util.*

class ClientsHandler(private val clientRepos: ClientRepository) {
    fun handleRegisterNewClient(request: ClientRequest): ClientResponse {
        val secRandom = SecureRandom()
        val bytes = ByteArray(32)
        secRandom.nextBytes(bytes)
        return clientRepos.createClient(
            secret = Base64.getEncoder().withoutPadding().encodeToString(bytes),
            user = request.user
        ).toResponse()
    }
}