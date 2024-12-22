package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.data.dao.Clients
import ac.osaka_u.ics.pbl.data.dao.Tasks
import ac.osaka_u.ics.pbl.data.entity.ErrorEntity
import kotlinx.datetime.Instant
import org.jetbrains.exposed.dao.id.EntityID
import org.jetbrains.exposed.sql.transactions.transaction
import java.util.*

interface ErrorRepository {
    fun create(taskId: UUID, stackTrace: String, timestamp: Instant, clientId: Int, assignmentId: UUID, clientVersion: String): Int
}

class ErrorRepositoryImpl : ErrorRepository {
    override fun create(taskId: UUID, stackTrace: String, timestamp: Instant, clientId: Int, assignmentId: UUID, clientVersion: String): Int {
        return transaction {
            ErrorEntity.new {
                task = EntityID(taskId, Tasks)
                this.stackTrace = stackTrace
                this.timestamp = timestamp
                client = EntityID(clientId, Clients)
                assignment = EntityID(assignmentId, Tasks)
                this.clientVersion = clientVersion
            }.id.value
        }
    }
}