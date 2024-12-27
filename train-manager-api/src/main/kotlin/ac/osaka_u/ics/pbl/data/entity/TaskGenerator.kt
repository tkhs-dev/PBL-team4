package ac.osaka_u.ics.pbl.data.entity

import ac.osaka_u.ics.pbl.data.dao.TaskGenerators
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import org.jetbrains.exposed.dao.IntEntity
import org.jetbrains.exposed.dao.IntEntityClass
import org.jetbrains.exposed.dao.id.EntityID

class TaskGeneratorEntity(id: EntityID<Int>) : IntEntity(id) {
    companion object : IntEntityClass<TaskGeneratorEntity>(TaskGenerators)

    var name by TaskGenerators.name
    var type by TaskGenerators.type
    var weight by TaskGenerators.weight

    var parameters by TaskGenerators.parameters
}

fun TaskGeneratorEntity.toModel() = TaskGenerator(
    id = id.value,
    name = name,
    type = type,
    weight = weight,
    parameters = parameters
)

fun TaskGenerator.toEntity() = TaskGeneratorEntity.new {
    name = this@toEntity.name
    type = this@toEntity.type
    weight = this@toEntity.weight
    parameters = this@toEntity.parameters
}