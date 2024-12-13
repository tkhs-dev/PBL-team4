package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.data.dao.Assignments
import ac.osaka_u.ics.pbl.data.dao.Clients
import ac.osaka_u.ics.pbl.domain.model.Assignment
import org.jetbrains.exposed.dao.UUIDEntity
import org.jetbrains.exposed.dao.UUIDEntityClass
import org.jetbrains.exposed.dao.id.EntityID
import java.util.*

class AssignmentEntity(id: EntityID<UUID>) : UUIDEntity(id) {
    companion object : UUIDEntityClass<AssignmentEntity>(Assignments)

    var assignedAt by Assignments.assignedAt
    var client by Assignments.client
    var deadline by Assignments.deadline
    var status by Assignments.status
    var statusChangedAt by Assignments.statusChangedAt
    var task by TaskEntity referencedOn Assignments.task
}

fun AssignmentEntity.toModel() = Assignment(
    id = id.value,
    clientId = client.value,
    assignedAt = assignedAt,
    deadline = deadline,
    status = status,
    statusChangedAt = statusChangedAt,
    task = task.toModel()
)

fun Assignment.toEntity() = AssignmentEntity.new(id) {
    assignedAt = this@toEntity.assignedAt
    client = EntityID(this@toEntity.clientId, Clients)
    deadline = this@toEntity.deadline
    status = this@toEntity.status
    statusChangedAt = this@toEntity.statusChangedAt
    task = this@toEntity.task.toEntity()
}