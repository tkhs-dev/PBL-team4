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

    var completed by Tasks.completed
    var baseModelId by Tasks.baseModelId
    var type by Tasks.type

    var parameters: Map<String, Any>
        get(){
            val jsonString = Tasks.parameters.getValue(this, TaskEntity::parameters)
            return Json.decodeFromString(ParameterMapSerializer, jsonString)
        }
        set(value) {
            val obj = Json.encodeToString(ParameterMapSerializer, value)
            Tasks.parameters.setValue(this, TaskEntity::parameters, obj)
        }
}

fun TaskEntity.toModel() = Task(
    id = id.value,
    completed = completed,
    baseModelId = baseModelId?.value,
    type = type,
    parameter = parameters
)

fun Task.toEntity() = TaskEntity.new(id) {
    completed = this@toEntity.completed
    baseModelId = this@toEntity.baseModelId?.let { EntityID(it, Models) }
    type = this@toEntity.type
    parameters = this@toEntity.parameter
}