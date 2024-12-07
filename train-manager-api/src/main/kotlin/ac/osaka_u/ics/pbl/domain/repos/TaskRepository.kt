package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.common.TaskType
import ac.osaka_u.ics.pbl.data.dao.Models
import ac.osaka_u.ics.pbl.data.entity.TaskEntity
import ac.osaka_u.ics.pbl.data.entity.toEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Task
import org.jetbrains.exposed.dao.id.EntityID
import org.jetbrains.exposed.sql.transactions.transaction
import java.util.*

class TaskUpdateBuilder {
    var completed: Boolean? = null
    var baseModelId: UUID? = null
    var type: TaskType? = null
    var parameter: Map<String, Any>? = null
}

interface TaskRepository {
    fun findTaskById(id: UUID): Task?
    fun findTasks(limit: Int = 10, completed: Boolean = false): List<Task>
    fun createTask(task: Task): Task
    fun updateTask(id: UUID, update: TaskUpdateBuilder.() -> Unit): Task?
    fun deleteTask(task: Task)
}

class TaskRepositoryImpl : TaskRepository {
    override fun findTaskById(id: UUID): Task? {
        return transaction {
            TaskEntity.findById(id)?.toModel()
        }
    }

    override fun findTasks(limit: Int, completed: Boolean): List<Task> {
        return transaction {
            TaskEntity.all().limit(limit).map { it.toModel() }
        }
    }

    override fun createTask(task: Task): Task {
        return transaction {
            task.toEntity().toModel()
        }
    }

    override fun updateTask(id: UUID, update: TaskUpdateBuilder.() -> Unit): Task? {
        val updateReq = TaskUpdateBuilder().apply(update)
        return transaction {
            TaskEntity.findById(id)?.apply {
                updateReq.completed?.let { completed = it }
                updateReq.baseModelId?.let { baseModelId = EntityID(it, Models) }
                updateReq.type?.let { type = it }
                updateReq.parameter?.let { parameters = it }
            }?.toModel()
        }
    }

    override fun deleteTask(task: Task) {
        transaction {
            TaskEntity.findById(task.id)?.delete()
        }
    }
}