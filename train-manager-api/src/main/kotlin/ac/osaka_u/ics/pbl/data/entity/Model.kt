package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.data.dao.Models
import ac.osaka_u.ics.pbl.domain.model.Model
import org.jetbrains.exposed.dao.UUIDEntity
import org.jetbrains.exposed.dao.UUIDEntityClass
import org.jetbrains.exposed.dao.id.EntityID
import java.util.*

class ModelEntity(id: EntityID<UUID>): UUIDEntity(id) {
    companion object : UUIDEntityClass<ModelEntity>(Models)

    var version by Models.version
    var parentId by Models.parentId
    var rootModel by Models.rootModel
    var sequence by Models.sequence
    var taskId by Models.taskId
    var createdAt by Models.createdAt
}

fun ModelEntity.toModel() = Model(
    id = id.value,
    version = version,
    parentId = parentId?.value,
    rootModel = rootModel?.value,
    sequence = sequence,
    taskId = taskId.value,
    createdAt = createdAt
)

fun Model.toEntity() = ModelEntity.new(id) {
    version = this@toEntity.version
    parentId = this@toEntity.parentId?.let { EntityID(it, Models) }
    rootModel = this@toEntity.rootModel?.let { EntityID(it, Models) }
    sequence = this@toEntity.sequence
    taskId = EntityID(this@toEntity.taskId, Models)
    createdAt = this@toEntity.createdAt
}