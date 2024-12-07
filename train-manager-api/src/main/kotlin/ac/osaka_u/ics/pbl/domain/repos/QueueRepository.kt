package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.data.entity.TaskEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Task
import org.jetbrains.exposed.sql.transactions.transaction
import java.util.*

interface QueueRepository {
    fun get(): List<Task>
    fun enqueue(taskId: UUID)
    fun dequeue(): Task?
    fun remove(taskId: UUID)
    fun clear()
}

class QueueRepositoryImpl : QueueRepository {
    private val queue = ArrayDeque<UUID>()

    override fun get(): List<Task> {
        return queue.mapNotNull {
            transaction {
                TaskEntity.findById(it)?.toModel()
            }
        }
    }

    override fun enqueue(taskId: UUID) {
        queue.add(taskId)
    }

    override fun dequeue(): Task? {
        return queue.poll()?.let {
            transaction {
                TaskEntity.findById(it)?.toModel()
            }
        }
    }

    override fun remove(taskId: UUID) {
        queue.remove(taskId)
    }

    override fun clear() {
        queue.clear()
    }
}