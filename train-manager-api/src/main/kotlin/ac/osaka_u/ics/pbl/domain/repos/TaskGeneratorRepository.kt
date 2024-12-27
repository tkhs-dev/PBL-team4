package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.common.TaskGeneratorType

import ac.osaka_u.ics.pbl.data.entity.TaskGeneratorEntity
import ac.osaka_u.ics.pbl.data.entity.toEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import org.jetbrains.exposed.sql.transactions.transaction

class TaskGeneratorUpdateBuilder{
    var name: String? = null
    var type: TaskGeneratorType? = null
    var weight: Int? = null
    var parameters: Map<String, Any>? = null
}

interface TaskGeneratorRepository{
    fun findTaskGenerators(): List<TaskGenerator>
    fun findTaskGeneratorById(id: Int): TaskGenerator?
    fun create(generator: TaskGenerator): TaskGenerator
    fun update(id: Int, update: TaskGeneratorUpdateBuilder.() -> Unit): TaskGenerator?
    fun delete(id: Int)
}

class TaskGeneratorRepositoryImpl : TaskGeneratorRepository{
    private val cache = mutableMapOf<Int, TaskGenerator>()

    override fun findTaskGenerators(): List<TaskGenerator> {
        return transaction {
            TaskGeneratorEntity.all().map { it.toModel() }
        }
    }

    override fun findTaskGeneratorById(id: Int): TaskGenerator? {
        return transaction {
            TaskGeneratorEntity.findById(id)?.toModel()
        }
    }

    override fun create(generator: TaskGenerator): TaskGenerator {
        return transaction{
            generator.toEntity().toModel()
        }
    }

    override fun update(id: Int, update: TaskGeneratorUpdateBuilder.() -> Unit): TaskGenerator? {
        val builder = TaskGeneratorUpdateBuilder().apply(update)
        return transaction {
            TaskGeneratorEntity.findById(id)?.apply {
                builder.name?.let { name = it }
                builder.type?.let { type = it }
                builder.weight?.let { weight = it }
                builder.parameters?.let { parameters = it }
            }?.toModel()
        }
    }

    override fun delete(id: Int) {
        transaction {
            TaskGeneratorEntity.findById(id)?.delete()
        }
    }
}

