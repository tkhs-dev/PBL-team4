package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.data.dao.Clients
import ac.osaka_u.ics.pbl.data.entity.ClientEntity
import ac.osaka_u.ics.pbl.data.entity.toEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Client
import org.jetbrains.exposed.sql.transactions.transaction

interface ClientRepository {
    fun findClientById(id: Int): Client?
    fun findClientBySecret(secret: String): Client?
    fun createClient(client: Client): Client
}

class ClientRepositoryImpl : ClientRepository {
    val cache = mutableMapOf<String,Int>()
    override fun findClientById(id: Int): Client? {
        return transaction {
            ClientEntity.findById(id)?.toModel()
        }
    }

    override fun findClientBySecret(secret: String): Client? {
        cache[secret]?.let { return findClientById(it) }
        return transaction {
            val client = ClientEntity.find { Clients.secret eq secret }.firstOrNull()?.toModel()
            client?.let { cache[secret] = it.id }
            client
        }
    }

    override fun createClient(client: Client): Client {
        return transaction{
            client.toEntity().toModel()
        }
    }
}