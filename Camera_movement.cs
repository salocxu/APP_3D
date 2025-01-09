using UnityEngine;

public class Camera_movement : MonoBehaviour
{
    public float speed = 10f;         // Vitesse de déplacement
    public float lookSpeed = 2f;     // Sensibilité de la souris

    private float yaw = 0f;          // Rotation horizontale
    private float pitch = 0f;        // Rotation verticale

    void Update()
    {
        // Mouvement global haut/bas (Touches E et Q par défaut)
        Vector3 globalMove = Vector3.zero;
        if (Input.GetKey(KeyCode.Space)) globalMove += Vector3.up * speed * Time.deltaTime;
        if (Input.GetKey(KeyCode.LeftShift)) globalMove += Vector3.down * speed * Time.deltaTime;

        // Mouvement local relatif à la caméra
        Vector3 localMove = Vector3.zero;
        if (Input.GetAxis("Vertical") != 0 || Input.GetAxis("Horizontal") != 0)
        {
            // Récupération des axes locaux de la caméra
            Vector3 forward = transform.forward;
            Vector3 right = transform.right;

            // Ignorer l'axe vertical pour rester dans le plan XZ
            forward.y = 0f;
            right.y = 0f;

            // Normalisation pour éviter les mouvements rapides
            forward.Normalize();
            right.Normalize();

            // Calcul du déplacement en fonction des inputs
            float moveForward = Input.GetAxis("Vertical") * speed * Time.deltaTime;
            float moveSide = Input.GetAxis("Horizontal") * speed * Time.deltaTime;

            localMove = (forward * moveForward) + (right * moveSide);
        }

        // Appliquer le déplacement global et local
        transform.position += globalMove + localMove;

        // Rotation de la caméra (Clic droit maintenu)
        if (Input.GetMouseButton(1)) // Clic droit maintenu
        {
            yaw += Input.GetAxis("Mouse X") * lookSpeed;
            pitch -= Input.GetAxis("Mouse Y") * lookSpeed;
            pitch = Mathf.Clamp(pitch, -90f, 90f); // Limite pour éviter les rotations absurdes

            transform.eulerAngles = new Vector3(pitch, yaw, 0f);
        }
    }
}
