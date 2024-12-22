package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.common.AssignmentStatus
import ac.osaka_u.ics.pbl.data.dao.Assignments
import ac.osaka_u.ics.pbl.data.dao.Clients
import ac.osaka_u.ics.pbl.data.dao.Tasks
import ac.osaka_u.ics.pbl.data.entity.AssignmentEntity
import ac.osaka_u.ics.pbl.data.entity.TaskEntity
import ac.osaka_u.ics.pbl.data.entity.toModel
import ac.osaka_u.ics.pbl.domain.model.Assignment
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import org.jetbrains.exposed.dao.id.EntityID
import org.jetbrains.exposed.sql.and
import org.jetbrains.exposed.sql.transactions.transaction
import org.jetbrains.exposed.sql.update
import java.util.*

class AssignmentUpdateBuilder{
    var assignedAt: Instant? = null
    var clientId: Int? = null
    var deadline: Instant? = null
    var status: AssignmentStatus? = null
    var statusChangedAt: Instant? = null
    var taskId: UUID? = null
}

interface AssignmentRepository {
    fun findAssignmentById(id: UUID): Assignment?
    fun findAssignmentByUserId(userId: Int): List<Assignment>
    fun findProcessingAssignmentsByUserId(userId: Int): List<Assignment>
    fun findAssignmentsShouldBeTimeout(): List<Assignment>
    fun findAssignments(): List<Assignment>
    fun createAssignment(
        assignedAt: Instant,
        clientId: Int,
        deadline: Instant,
        status: AssignmentStatus,
        statusChangedAt: Instant,
        taskId: UUID
    ): Assignment
    fun updateAssignment(id: UUID, update: AssignmentUpdateBuilder.() -> Unit): Assignment?
    fun deleteAssignment(assignment: Assignment)
}

class AssignmentRepositoryImpl : AssignmentRepository {
    override fun findAssignmentById(id: UUID): Assignment? {
        return transaction {
            AssignmentEntity.findById(id)?.toModel()
        }
    }

    override fun findAssignmentByUserId(userId: Int): List<Assignment> {
        return transaction {
            AssignmentEntity.find { Assignments.client eq userId }.map { it.toModel() }
        }
    }

    override fun findProcessingAssignmentsByUserId(userId: Int): List<Assignment> {
        return transaction {
            AssignmentEntity.find { (Assignments.client eq userId) and (Assignments.status eq AssignmentStatus.PROCESSING) }.map { it.toModel() }
        }
    }

    override fun findAssignmentsShouldBeTimeout(): List<Assignment> {
        return transaction {
            AssignmentEntity.find { Assignments.deadline less Clock.System.now() and (Assignments.status eq AssignmentStatus.PROCESSING) }.map { it.toModel() }
        }
    }

    override fun findAssignments(): List<Assignment> {
        return transaction {
            AssignmentEntity.all().map { it.toModel() }
        }
    }

    override fun createAssignment(
        assignedAt: Instant,
        clientId: Int,
        deadline: Instant,
        status: AssignmentStatus,
        statusChangedAt: Instant,
        taskId: UUID
    ): Assignment {
        return transaction {
            AssignmentEntity.new {
                this.assignedAt = assignedAt
                this.client = EntityID(clientId, Clients)
                this.deadline = deadline
                this.status = status
                this.statusChangedAt = statusChangedAt
                this.task = TaskEntity[taskId]
            }.toModel()
        }
    }

    override fun updateAssignment(id: UUID, update: AssignmentUpdateBuilder.() -> Unit): Assignment? {
        val builder = AssignmentUpdateBuilder().apply(update)
        transaction {
            Assignments.update({Assignments.id eq id}) {
                builder.assignedAt?.let { assignedAt -> it[Assignments.assignedAt] = assignedAt }
                builder.clientId?.let { clientId -> it[Assignments.client] = EntityID(clientId, Clients) }
                builder.deadline?.let { deadline -> it[Assignments.deadline] = deadline }
                builder.status?.let { status -> it[Assignments.status] = status }
                builder.statusChangedAt?.let { statusChangedAt -> it[Assignments.statusChangedAt] = statusChangedAt }
                builder.taskId?.let { taskId -> it[Assignments.task] = EntityID(taskId, Tasks) }
            }
        }
        return findAssignmentById(id)
    }

    override fun deleteAssignment(assignment: Assignment) {
        transaction {
            AssignmentEntity.findById(assignment.id)?.delete()
        }
    }
}