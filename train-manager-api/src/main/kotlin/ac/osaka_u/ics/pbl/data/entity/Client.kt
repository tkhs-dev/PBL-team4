package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.data.dao.Clients
import ac.osaka_u.ics.pbl.domain.model.Client
import org.jetbrains.exposed.dao.IntEntity
import org.jetbrains.exposed.dao.IntEntityClass
import org.jetbrains.exposed.dao.id.EntityID

class ClientEntity(id: EntityID<Int>) : IntEntity(id) {
    companion object : IntEntityClass<ClientEntity>(Clients)

    var secret by Clients.secret
}

fun ClientEntity.toModel() = Client(
    id = id.value,
    secret = secret
)

fun Client.toEntity() = ClientEntity.new {
    secret = this@toEntity.secret
}