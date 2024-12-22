package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.common.ParameterMapSerializer
import ac.osaka_u.ics.pbl.data.dao.Models
import ac.osaka_u.ics.pbl.data.dao.Tasks
import ac.osaka_u.ics.pbl.domain.model.Task
import kotlinx.serialization.json.Json
import org.jetbrains.exposed.dao.UUIDEntity
import org.jetbrains.exposed.dao.UUIDEntityClass
import org.jetbrains.exposed.dao.id.EntityID
import java.util.*

class TaskEntity(id: EntityID<UUID>) : UUIDEntity(id) {
    companion object : UUIDEntityClass<TaskEntity>(Tasks)

    var status by Tasks.status
    var errorCount by Tasks.errorCount
    var baseModelId by Tasks.baseModelId
    var type by Tasks.type
    var createdAt by Tasks.createdAt
    var parameters by Tasks.parameters
}

fun TaskEntity.toModel() = Task(
    id = id.value,
    errorCount = errorCount,
    status = status,
    baseModelId = baseModelId?.value,
    type = type,
    createdAt = createdAt,
    parameter = parameters
)

fun Task.toEntity() = TaskEntity.new(id) {
    status = this@toEntity.status
    errorCount = this@toEntity.errorCount
    baseModelId = this@toEntity.baseModelId?.let { EntityID(it, Models) }
    type = this@toEntity.type
    createdAt = this@toEntity.createdAt
    parameters = this@toEntity.parameter
}